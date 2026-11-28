# Backward screening before LP tightening on WK17a

We tested the public fixed-budget feasibility call with one target and nine non-target classes on
the first eight MNIST test images. The target was the digit following the true digit, wrapping 9
to 0. Each pixel could change by ±0.1 within [0, 1]. The target needed a score lead of at least
0.125 over every non-target. Both settings ruled out every target in both repeats. Screening reduced
solver models from eight to zero per repeat.

| Image     | Screening off, mean seconds | Screening on, mean seconds |
| --------- | --------------------------: | -------------------------: |
| 1         |                       0.796 |                      0.111 |
| 2         |                       0.838 |                      0.072 |
| 3         |                       0.564 |                      0.063 |
| 4         |                       0.823 |                      0.063 |
| 5         |                       0.783 |                      0.071 |
| 6         |                       0.546 |                      0.063 |
| 7         |                       0.768 |                      0.292 |
| 8         |                       0.589 |                      0.070 |
| **Total** |                   **5.707** |                  **0.803** |

The summed per-image time fell by 85.9% on this cohort. Repeat totals were 6.387 and 5.028 s with
screening off, and 0.968 and 0.639 s with it on. The larger within-setting spread, divided by that
setting’s mean, gives a ±40.9 percentage-point repeat-noise allowance. All eight paired means
decreased. The median per-image ratio was 0.114. These images all finished at the bounding stage, so
the size of the saving is specific to this workload. [Per-call data](first-eight.csv).

### An image requiring the solver

We separately chose WK17a image 480, a handwritten 9, because it required the solver in the earlier
benchmark. We checked whether digit 8 could tie or win, using zero margin. Both settings built a
model on both repeats. With screening off, both calls found a verified witness (3.60 s and 5.40 s).
With screening on, one call timed out without a witness (6.37 s) and the other returned a verified
witness (6.17 s). End-to-end time includes work before the final solve. This single image
illustrates the cost of an inconclusive screen and variation near a solver time limit.
[Per-call data](fallback.csv).

### Reuse of backward bounds during LP and MIP tightening

On the same image 480 and target digit 8, we compared the previous #335 implementation (`bbf0b1e`)
with the version that passes backward ranges to LP and MIP tightening. Both versions ran the
backward screen. The earlier version discarded its ranges before constructing these solver models.
We timed each public API call twice after a warm-up, pinned to the same CPU with identical options.
Each version used a 0.2-second limit per tightening solve and five seconds for the final solve.

| Tightening                    | Bound solves before → after | Mean formulation time before → after | Mean end-to-end time before → after |
| ----------------------------- | --------------------------: | -----------------------------------: | ----------------------------------: |
| LP                            |                   202 → 198 |                        0.65 → 0.80 s |                       6.46 → 6.42 s |
| MIP (including its LP solves) |                   398 → 394 |                        3.98 → 3.62 s |                       9.92 → 9.94 s |

Reusing the ranges saved four bound solves on every repeat. Both versions left 151 ReLUs with
undetermined signs after solver tightening. Each version found a checked witness in one of its two
LP runs and both MIP runs. The other LP run reached the time limit. The end-to-end time reduction
was 0.5 ± 14.7 percentage points for LP and −0.3 ± 3.6 for MIP. Each allowance is the larger
within-version repeat spread divided by that version’s mean. This one image establishes the
reduction in bound solves. Its timings do not resolve an end-to-end speed difference.
[Per-call data](fallback-bound-reuse.csv).

To reproduce, run [the same driver](reuse_bounds.jl) sequentially against each checkout’s benchmark
environment on the same CPU. Pass an output CSV path as the only argument. Repeat zero warms the
method and is excluded from the table.

### Pass the surviving targets to the solver

On WK17a image 480, the backward score-gap bounds ruled out eight of the nine rivals of digit 9.
Digit 8 remained. We compared rebuilding the original target list (`942d698`) with passing just that
surviving target to the solver. Both versions used the same backward neuron bounds. The public
feasibility calls used the same single-threaded solver, CPU, pixel budget and time limits as the
bound-reuse comparison above. Two timed repeats followed one warm-up for each method.

| Tightening          | Bound solves before → after | Binary variables before → after | Structural constraints before → after | Mean end-to-end time before → after |
| ------------------- | --------------------------: | ------------------------------: | ------------------------------------: | ----------------------------------: |
| Interval arithmetic |                       0 → 0 |                       159 → 155 |                         1,414 → 1,405 |                       5.61 → 5.58 s |
| Target LP           |                       4 → 1 |                       155 → 155 |                         1,406 → 1,406 |                       6.73 → 6.64 s |

Both interval-arithmetic repeats reached the five-second final solve limit under both versions. All
four Target LP runs found a checked witness. Reusing the target list removed four binary variables
in interval-arithmetic mode and three bound solves in Target LP on each repeat. The end-to-end
reductions were 0.5 ± 2.4 percentage points for interval arithmetic and 1.3 ± 8.1 for Target LP. The
allowances use the larger within-version repeat spread divided by that version’s mean. Neither time
difference clears repeat variation on this image. To reproduce, run the [driver](reuse_bounds.jl) on
each checkout with an output CSV path followed by `all-rivals`.
[Per-call data](partial-target-pruning.csv).

### Reproduce

We used the [pinned environment](../2026-11-28-public-default/README.md#reproduce) on the same Intel
Core i7-8700K. Both settings used code from the follow-up to #327. Screening off exercises the
previous LP-tightened JuMP formulation for this target selection and margin. The only option changed
within each pair was `backward_screen`. HiGHS used one thread, two seconds per tightening solve and
five seconds for the final MIP. Image 9 warmed each mode before measurement. Each image’s two
methods and two repeats ran on CPU 2, with the order reversed on repeat two. Times are the public
API’s `:TotalTime`. Run these commands sequentially from the repository root:

```sh
probe=benchmarks/reports/2026-11-28-screen-before-tightening/run.jl
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 2 julia --project=benchmarks "$probe" /tmp/first-eight.csv
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 2 julia --project=benchmarks "$probe" /tmp/fallback.csv fallback
```
