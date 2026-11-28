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
