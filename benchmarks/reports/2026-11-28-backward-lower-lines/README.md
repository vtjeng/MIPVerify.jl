# Parallel lower lines in the backward verifier

The current adaptive lower line certified more images directly on both MNIST networks at a pixel
budget of ±0.10. Keep the adaptive rule as the default. The parallel rule increased solver referrals
on both cohorts, showed no repeatable speed advantage on WK17a, and was slightly slower on both
RSL18a repeats. Both rules returned the same per-image outcomes in both repeats.

| Cohort and rule  | Direct certificates | Solver referrals | Time, repeat 1 (s) | Time, repeat 2 (s) | Mean total (s) | Mean screening time (s) | Mean final MIP time (s) |
| ---------------- | ------------------: | ---------------: | -----------------: | -----------------: | -------------: | ----------------------: | ----------------------: |
| WK17a, adaptive  |               1,844 |               34 |              498.7 |              508.1 |          503.4 |                   238.0 |                   232.7 |
| WK17a, parallel  |               1,833 |               42 |              516.3 |              501.1 |          508.7 |                   240.8 |                   234.1 |
| RSL18a, adaptive |                  35 |               49 |            5,529.8 |            5,523.3 |        5,526.6 |                     1.5 |                 5,425.0 |
| RSL18a, parallel |                  29 |               55 |            5,552.4 |            5,547.2 |        5,549.8 |                     1.9 |                 5,434.9 |

### WK17a

Across 2,000 images, the parallel rule changed mean total time by **+1.05%** with a **±2.99
percentage point repeat-noise allowance**. The changes in the two individual repeats were +3.54% and
-1.39%. The allowance uses the larger within-rule difference between repeat totals, divided by that
rule’s mean. It describes repeatability on this fixed cohort.

The public verifier screened 17,559 rivals after the original-input shortcut. The adaptive rule
ruled out 17,430 and the parallel rule ruled out 17,414. The parallel line gave a strictly tighter
numerical upper bound on 1,858 rivals. Only adaptive directly certified 13 images and only parallel
directly certified 2. Taking the tighter bound for each rival across the two calculations would
certify 1,846 images. Timings here measure each rule separately.

Across per-image mean times, the parallel/adaptive ratios at the 10th, 50th and 90th percentiles
were 0.360, 1.010 and 2.900. Parallel was faster by more than 1% on 920 images, within 1% on 80, and
slower by more than 1% on 1,000.

| Rule     | Original witnesses | Corner witnesses | PGD witnesses | Bound LP calls (repeat 1 / 2) | Robust (repeat 1 / 2) | Witness (repeat 1 / 2) | Unresolved (repeat 1 / 2) |
| -------- | -----------------: | ---------------: | ------------: | ----------------------------: | --------------------: | ---------------------: | ------------------------: |
| adaptive |                 49 |               68 |             5 |                     104 / 104 |           1874 / 1874 |              126 / 126 |                     0 / 0 |
| parallel |                 49 |               74 |             2 |                     125 / 125 |           1874 / 1874 |              126 / 126 |                     0 / 0 |

### RSL18a

Across 100 images, the parallel rule changed mean total time by **+0.42%** with a **±0.12 percentage
point repeat-noise allowance**. The changes in the two individual repeats were +0.41% and +0.43%.
The allowance uses the larger within-rule difference between repeat totals, divided by that rule’s
mean. It describes repeatability on this fixed cohort.

The public verifier screened 882 rivals after the original-input shortcut. The adaptive rule ruled
out 554 and the parallel rule ruled out 484. The parallel line gave a strictly tighter numerical
upper bound on 0 rivals. Only adaptive directly certified 6 images and only parallel directly
certified 0. Taking the tighter bound for each rival across the two calculations would certify 35
images. Timings here measure each rule separately.

Across per-image mean times, the parallel/adaptive ratios at the 10th, 50th and 90th percentiles
were 0.875, 1.002 and 1.386. Parallel was faster by more than 1% on 19 images, within 1% on 47, and
slower by more than 1% on 34.

| Rule     | Original witnesses | Corner witnesses | PGD witnesses | Bound LP calls (repeat 1 / 2) | Robust (repeat 1 / 2) | Witness (repeat 1 / 2) | Unresolved (repeat 1 / 2) |
| -------- | -----------------: | ---------------: | ------------: | ----------------------------: | --------------------: | ---------------------: | ------------------------: |
| adaptive |                  2 |               12 |             2 |                     441 / 441 |               42 / 42 |                16 / 16 |                   42 / 42 |
| parallel |                  2 |               12 |             2 |                     495 / 495 |               42 / 42 |                16 / 16 |                   42 / 42 |

The [previous hosted RSL18a run](../2026-11-28-public-default/rsl18a/README.md) certified one more
image with the adaptive rule. Image 51 reached the 120-second limit locally under both lower lines.
All timing comparisons above use the fresh local pairs.

## Why the slope helps some targets

On an interval $l<0<u$, the largest vertical error of the line $h=\alpha z$ is the larger of
$-\alpha l$ and $(1-\alpha)u$. The parallel choice $\alpha=u/(u-l)$ makes the two endpoint errors
equal. The backward calculation combines many signed contributions before maximizing over a shared
pixel box. Balancing one neuron’s worst error can therefore improve or worsen the final upper bound.

WK17a image 544 illustrates the benefit. The upper bound for rival 9 went from +0.004780 with the
adaptive line to −0.09293 with the parallel line, enough to certify the image directly. On image 19
the upper bound for rival 5 went from −0.2435 to +0.02985, sending that image to the solver.

## Method

The experiment compares the lower line used by back-substitution on MNIST test images 1–2,000 for
WK17a and 1–100 for RSL18a. Each image keeps its original MNIST test index. The pixel budget is
±0.10 clipped to [0, 1]. The pinned environment contains Julia 1.12.6, HiGHS.jl 1.23.0 and HiGHS
1.14.0.

For a ReLU whose input interval crosses zero, both rules use the upper chord. For a negative
coefficient in an upper-bound calculation, the adaptive rule uses the diagonal lower line when the
positive side is at least as long as the negative side. Otherwise it uses the horizontal lower line.
The parallel rule instead uses the line through the origin with slope $u/(u-l)$ for that neuron's
current input interval $[l,u]$. Stable ReLUs use the exact line. Both methods back-substitute
through the same layers, including when computing intermediate hidden-neuron ranges. Each method
supplies its own computed ranges to its solver fallback.

Both configurations call the public `find_adversarial_example` in feasibility mode with the
original-input shortcut enabled. Target LP receives 20 seconds per bound solve and the final MIP
receives 120 seconds. The two settings share the code and pinned environment of PR #327 at
`0f737658c36e9af07a19b433e9b2f24dd35247e6`. The [experimental patch](measure.patch) adds a switch
inside the backward ReLU routine and records the time and bounds of the screen. All paths use the
same corner check, PGD and fallback implementation.

Five single-threaded Julia workers used distinct physical cores (logical CPUs 0, 2, 4, 6 and 8) on
an Intel Core i7-8700K. For every image both settings ran twice on the same core. Their order
alternated by image and reversed on the second repeat. Each worker warmed both variants on WK17a
images 19, 63 and 66 before the WK17a timings, and RSL18a images 1 and 19 before the RSL18a timings.
Times sum the public API's per-image execution times and average the two sums for each network. They
exclude warm-up and writing CSVs. The screen-only time covers network compilation, hidden ranges and
target bounds. End-to-end time also covers search, solving and witness checking. The comparison uses
the same two networks as the [screening on/off study](../2026-11-28-public-default/README.md).

The [earlier forward experiment](../2026-11-27-full-verifier.md) added a second affine calculation
after adaptive backward screening. Here the lower line is replaced within the original backward
calculation.

## Reproduce

In a scratch checkout containing this report, apply the experiment patch, copy the archived Julia
environment to the repository root, and instantiate the benchmark project. The cached MNIST data and
weights must be available to all workers. Use an empty directory for the measurements.

```sh
study=benchmarks/reports/2026-11-28-backward-lower-lines
git apply "$study/measure.patch"
cp benchmarks/reports/2026-11-28-public-default/environment/Manifest.toml Manifest.toml
julia --project=benchmarks -e 'using Pkg; Pkg.instantiate(); using MIPVerify; for n in ("WK17a", "RSL18a"); MIPVerify.get_example_network_params("MNIST.$(n)_linf0.1_authors"); end; MIPVerify.read_datasets("MNIST")'
mkdir lower-line-runs
for lane in 0 1 2 3 4; do
  OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c "$((2 * lane))" \
    julia --project=benchmarks "$study/run.jl" "$lane" 5 lower-line-runs &
done
wait
python3 "$study/analyze.py" lower-line-runs
```

The [analysis](analyze.py) accepts raw or compressed CSVs. It checks each network, image, rule,
repeat and lane, verifies the relationship between the public witness flags and the reported
outcomes, and compares the numerical bounds between repeats. The [per-image records](runs/) contain
the timing breakdown, verifier route, outcome, bound-solver call count and all nine rival bounds for
each screened image. Original-input shortcuts skip the bound calculation.
