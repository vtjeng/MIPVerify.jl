# Check the affine corner before PGD

This historical experiment used the
[verifier and replay driver at the measured revision](https://github.com/vtjeng/MIPVerify.jl/blob/fae89161ee3aec9cfb630f2079de94d63d3400cc/benchmarks/reports/corner_search.jl).
Check out that revision to reproduce it.

We measured witness search on every image left open by the two affine screens in the
[screen-order comparison](2026-11-27-affine-order.md). Both versions checked the same MNIST images,
used the same surviving targets and pixel budget, and sent exactly the same images to the solver.
The candidate first evaluates the actual network at each surviving rival’s backward-bound corner and
caches the exact inward-rounded endpoints for the 256 MNIST brightness values. Both versions
validate any returned witness.

| Network | Images entering search | Corner / PGD / solver | PGD first (s), runs 1 / 2 | Corner first (s), runs 1 / 2 | Reduction, runs 1 / 2 |
| ------- | ---------------------: | --------------------: | ------------------------: | ---------------------------: | --------------------: |
| WK17a   |                    154 |          117 / 5 / 32 |               6.32 / 6.45 |                  5.35 / 5.20 |         15.4% / 19.4% |
| RSL18a  |                     65 |           14 / 2 / 49 |             19.81 / 19.75 |                20.41 / 19.92 |         -3.1% / -0.8% |

On WK17a, the estimated reduction was 17.4% ± 1.9 percentage points. Across images, 130 of 154
faster, with 10th, 50th and 90th percentile savings of -5.0, 7.8 and 13.5 ms.

On RSL18a, search time increased by 2.0% ± 0.3 percentage points. Across images, 29 of 65 faster,
with 10th, 50th and 90th percentile savings of -39.4, -2.9 and 14.2 ms.

The ± values give the relative difference between the two PGD-first totals, using their mean as the
denominator. They describe repeat variability. Per-image savings average both repeats. Timers cover
the search and witness validation after screening. They exclude computing the bounds, retaining the
corner directions during screening, and solving any remaining MIPs. These measurements establish how
often the verifier can skip PGD. End-to-end time requires a separate measurement.

We compared the parent verifier at `63e5d84` with the implementation accompanying this report, on
one pinned CPU with single-threaded BLAS (Intel Core i7-8700K). For each network we warmed both
paths, ran both methods twice on each image and reversed their order on the repeat.
[Per-image measurements](2026-11-27-corner-search.csv.gz).

```sh
mkdir -p /tmp/corner-parent
git archive 63e5d84 benchmarks | tar -x -C /tmp/corner-parent
OPENBLAS_NUM_THREADS=1 taskset -c 4 julia --project=benchmarks benchmarks/reports/corner_search.jl corner-search.csv /tmp/corner-parent/benchmarks
```
