# PGD full-start benchmark report

Date: 2026-07-20  
Implementation commit: `f7e2bec107f4155b84a6d69a04cfda9be950217b`

## Research idea

A verification run can stop as soon as it finds an allowed perturbation that changes the model's
predicted class. Otherwise, the solver must prove that no such perturbation exists. Projected
gradient descent (PGD) can search cheaply for these attacks, but it cannot provide that proof.

This experiment asks whether PGD remains useful when its attack fails. Among all points visited
during its search, PGD retains the one that comes closest to changing the prediction. Completing
that candidate into an initial value for every mixed-integer programming variable gives the solver a
strong feasible incumbent. Such an incumbent might reduce branch-and-bound work by allowing more of
the search tree to be pruned.

PGD and the verifier measure progress using the same classification margin. For an input with true
class `y`, they maximize:

```text
max_{x' in the L-infinity box} max_{j != y} f_j(x') - f_y(x').
```

Here, `f_k(x')` is the network's logit for class `k` at `x'`. A nonnegative margin is an adversarial
example and ends verification; a negative margin closer to zero is a stronger unsuccessful
candidate.

The analyzed cohort contained only samples for which neither PGD nor the random control found a
verified attack.

The experiment tests that hypothesis by comparing `pgd_full` (a completed PGD start) with `cold` (no
start) and `random_full` (a completed random start). The random control helps separate candidate
quality from the effect of supplying a complete start at all.

## Conclusion

The PGD warm start did not reduce verification work in this benchmark. Across all 12 selected
samples, `pgd_full` used 6.8% more simplex iterations than `cold`. In a separate check of the four
samples with the highest median `cold` simplex-iteration counts, the increase was 21.6%.

The 95% bootstrap intervals for the 12-sample simplex-iteration and end-to-end time ratios were both
entirely above 1.0. Including the time required to run PGD and construct the complete solver start,
`pgd_full` took 70.5% longer end to end than `cold`.

`random_full` controls for the effect of supplying any complete solver start, independent of PGD
candidate quality. Unlike `pgd_full`, it uses an unoptimized random input. At the cohort level,
`random_full` showed no clear simplex-work penalty relative to `cold`. This weighs against a generic
complete-start penalty as the main explanation for the `pgd_full` regression, but does not rule it
out: the confidence interval includes a penalty comparable to the `pgd_full` estimate, and
`random_full` timed out on sample 246 in all three repetitions.

PGD produced a better margin than the random candidate on all 12 samples. Even so, compared with
`cold`, `pgd_full` required more simplex work on five samples and the same amount on seven; it
improved none. Better candidate margins therefore did not translate into less branch-and-bound work.

## Method

### Configuration

- The benchmark used the MNIST WK17a neural network, an L-infinity perturbation radius of `0.1`, and
  linear programming (LP) bound tightening.
- HiGHS 1.14.x, an open-source LP and mixed-integer programming (MIP) solver, used one thread,
  parallel mode off, solver seed zero, and a 30-second solve limit.
- PGD used 20 restarts, 100 steps, step size `0.01`, and base seed `20260720`.
- The software versions were Julia 1.12.6, HiGHS.jl 1.23.0, and HiGHS_jll 1.14.0+0.
- The candidate cache SHA-256 was
  `f515fe0bf78e515344b3a2a3c7408e3362ba4a2f8acbfcd67537e0d55c2f6640`.

### Cohort selection

The cohort indices were chosen before any warm-start treatment results were inspected. The hard pool
comprised the slowest historical LP-feasibility cases after known attacks were excluded; the control
pool came from a class-balanced historical cohort. The selector retained correctly classified inputs
whose PGD candidates had independently verified negative margins. It took the first eight eligible
hard samples from the predeclared order. The other four samples were difficulty controls: they were
not selected for high historical solver cost, so they checked whether the result was confined to the
deliberately oversampled difficult cases. The selector preferred distinct true classes when choosing
these four controls.

After cohort selection, the benchmark generated one reproducible random candidate per sample. It
sampled each input coordinate uniformly between that coordinate's lower and upper limits in the
L-infinity perturbation box, using a recorded random seed. The random candidate was not optimized or
ranked by its margin. The benchmark checked both the PGD and random candidates by evaluating the
network and confirming that the candidate remained inside the allowed perturbation box. If either
candidate had changed the predicted class, the benchmark would have recorded the candidate as a
verified attack and skipped the MIP solves because the verification question had already been
answered. Neither candidate was an attack for any of the 12 selected samples.

All PGD and random candidate margins in the analyzed cohort were negative; among negative margins,
closer to zero is stronger.

### Treatments and repetitions

For each sample, the benchmark built and tightened one base MIP model. It then ran each treatment
three times. Every run used a fresh copy of the same tightened model, and the copies were created
and solved one at a time rather than in parallel. Treatment order rotated across the three
repetition blocks:

- `cold` supplied no start.
- `random_full` supplied the random candidate described above, projected it slightly inward, and
  completed it to every MIP variable.
- `pgd_full` supplied the highest-margin point across all steps of 20 PGD restarts, projected it
  slightly inward, and used the same completion to every MIP variable.

### Metrics

The primary metric is the median simplex-iteration count across the three repetitions for each
sample. Each simplex-iteration or node-count ratio is `(numerator + 1) / (denominator + 1)`; adding
one keeps zero-count cases defined. Aggregate ratios are geometric means over all 12 samples. Their
95% intervals come from resampling samples, and values above 1 indicate more work or time in the
numerator treatment. End-to-end time charges candidate generation and full-start completion to the
corresponding treatment. The complete protocol and decision thresholds are in
[README.md](README.md#decision-rules-and-definition-of-done).

## Fixed-cohort results

| Metric             |    `pgd_full / cold` | `random_full / cold` | `pgd_full / random_full` |
| ------------------ | -------------------: | -------------------: | -----------------------: |
| Simplex iterations | 1.068 [1.006, 1.160] | 1.013 [0.968, 1.070] |     1.055 [0.965, 1.170] |
| Nodes              | 0.911 [0.647, 1.203] | 0.685 [0.351, 1.099] |     1.329 [0.881, 2.605] |
| Main solve time    | 1.059 [0.938, 1.224] | 1.018 [0.889, 1.224] |     1.040 [0.874, 1.283] |
| End-to-end time    | 1.705 [1.457, 1.982] | 1.063 [0.950, 1.250] |     1.603 [1.253, 1.994] |

Intervals are 95% bootstrap intervals. All 36 `cold` and 36 `pgd_full` runs certified robustness.
`random_full` certified in 33 runs and reached the common time limit on sample 246 in all three
repetitions. No candidate or treatment found a verified attack. In all 36 `random_full` and 36
`pgd_full` runs, HiGHS reported the supplied start as feasible, and the trace observed it as the
initial incumbent.

| Sample | Stratum | PGD margin | Random margin | `cold` simplex | `random_full` simplex | `pgd_full` simplex | `pgd_full / cold` |
| -----: | ------- | ---------: | ------------: | -------------: | --------------------: | -----------------: | ----------------: |
|     19 | hard    |     -1.040 |        -3.316 |          2,648 |                 2,648 |              2,648 |             1.000 |
|     32 | control |     -4.636 |        -5.119 |            108 |                   108 |                108 |             1.000 |
|     36 | control |     -8.972 |       -10.421 |            242 |                   242 |                242 |             1.000 |
|     46 | hard    |     -0.982 |        -3.576 |         33,491 |                36,417 |             46,667 |             1.393 |
|    233 | hard    |     -0.674 |        -2.915 |            652 |                   652 |                660 |             1.012 |
|    246 | hard    |     -1.138 |        -3.550 |         34,290 |               32,745* |             36,541 |             1.066 |
|    313 | control |     -8.257 |        -9.834 |            285 |                   285 |                285 |             1.000 |
|    359 | hard    |     -0.390 |        -2.441 |         15,660 |                20,440 |             16,236 |             1.037 |
|    407 | hard    |     -0.787 |        -2.104 |         28,058 |                24,123 |             39,864 |             1.421 |
|    444 | hard    |     -0.314 |        -1.761 |            609 |                   609 |                609 |             1.000 |
|    460 | control |     -7.924 |        -9.590 |            220 |                   220 |                220 |             1.000 |
|    479 | hard    |     -0.766 |        -2.644 |          1,211 |                 1,211 |              1,211 |             1.000 |

`*` The displayed sample 246 `random_full` simplex count is the median work accumulated by the
30-second cutoff across three runs. `random_full` remained unresolved with a positive final
objective bound in all three runs; `cold` and `pgd_full` each certified in all three runs.

## Difficult-case and variance checks

The four samples with the highest median `cold` simplex-iteration counts in the fixed cohort were,
in descending order, 246, 46, 407, and 359. They formed the later difficult-case check. Two fresh
Julia processes ran them. The forward treatment order was `cold`, `random_full`, `pgd_full`; the
reverse order was `pgd_full`, `random_full`, `cold`. Every sample/treatment pair that resolved in
both processes reproduced the same simplex and node counters. The `pgd_full / cold` simplex ratio
was 1.216 with a 95% interval of [1.051, 1.407]. The `random_full / cold` ratio was 1.034 in the
forward run and 1.046 in the reverse run; the difference came from the time-limited sample 246.

The main cohort supplied three repetitions per sample. All work counters were identical within a
sample and treatment except:

- sample 46 `pgd_full`: two runs completed with 46,667 simplex iterations and 86 nodes; the third
  reached the time boundary and certified with 45,563 iterations and 25 nodes; and
- sample 246 `random_full`: all runs timed out, with 32,122 to 32,922 simplex iterations.

Wall times varied even when exact work counters matched. The comparable historical and fresh `cold`
runs were both collected after LP-certificate variable contributions were put in deterministic
order. Both sets covered samples 19, 32, 36, and 46. Fresh-to-historical ratios of median main-solve
wall times ranged from 0.916 to 1.009, within the predeclared `[0.5, 2.0]` no-flag range. The fresh
hard-tail processes also reproduced the `cold` work counters exactly.

Peak recorded process resident set size (RSS) was 2,130 MiB. The lowest recorded system-available
memory was 11,016 MiB, above the 4 GiB guard. Runs were serial, and no memory guard or Linux
control-group (cgroup) out-of-memory (OOM) event triggered.

## Validation

- Package suite: 884 passed, one test marked broken, zero failures.
- Benchmark helper suite: 108 passed.
- Focused PGD warm-start suite: 46 passed.
- Julia formatting check: passed.
- Result audit: 12 sample rows, 108 treatment rows, three repetitions for every sample/treatment, 72
  of 72 supplied starts used, and identical copied-model signatures within each sample.

## Interpretation

This non-monotonic behavior is known in branch-and-bound search. Ragsdale and Shapiro reported that
a better initial incumbent can produce a longer search and cautioned that incumbent use is heuristic
([paper abstract](https://www.sciencedirect.com/science/article/pii/0305054895000372)). HiGHS uses a
feasible MIP start to set the primal upper bound and the objective threshold used by the node queue;
these values can change the subsequent search
([HiGHS 1.14 source](https://github.com/ERGO-Code/HiGHS/blob/v1.14.0/highs/mip/HighsMipSolverData.cpp#L793-L831)).

The results are consistent with that mechanism. A complete start supplies a discrete ReLU and
selector assignment in addition to a scalar margin. PGD improves the scalar margin, but its induced
discrete assignment can steer cuts, heuristics, pruning, and node selection onto a worse path. This
is an inference from the solver mechanism and observed counters; the benchmark does not isolate
which individual HiGHS decision caused each regression.

Diagnostic sparse-start results are not part of the primary comparison. Partial starts make HiGHS
run a completion procedure, which is a separate source of work
([HiGHS MIP-start documentation](https://ergo-code.github.io/HiGHS/stable/guide/further/)). Both
completed starts, `random_full` and `pgd_full`, hold that completion-interface difference constant.

## Limitations

- The cohort covers one network, perturbation radius, solver generation, and fixed solver seed.
- Three order-rotated repetitions and two fresh hard-tail processes provide checks on timing and
  fixed-limit variance under the fixed solver seed; variation across solver seeds remains
  unmeasured. Gurobi's performance guidance recommends comparing distributions over multiple seeds
  for a general solver claim
  ([staff guidance](https://support.gurobi.com/hc/en-us/community/posts/39029910894353-Why-is-it-slower-to-enter-MIP-start-than-not-to-enter-it)).
- Seven of 12 samples solve at the root with identical work across treatments. The hard-tail result
  is more informative about branch-and-bound behavior but has only four samples.
- The random control is one deterministic draw per sample. Multiple draws could better characterize
  the distribution of discrete start quality. The predeclared rejection depends on
  `pgd_full / cold`, so additional random draws are outside that decision.

Under the predeclared definition of done, the proposed `pgd_full` warm start is **not supported** at
the solver level and **not practical** end to end.
