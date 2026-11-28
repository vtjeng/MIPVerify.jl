# Public feasibility verification on 100 RSL18a images

This comparison calls `MIPVerify.find_adversarial_example` on MNIST test images 1–100 with the
RSL18a network. Both configurations use the same production implementation at
`26e928a423aab28df03448b6f0aad37ad9f1650d`. The candidate enables the public `backward_screen`
default, and the baseline disables it. Both keep the original-input shortcut.

Total time for 100 images, averaged over two repeats, fell from 7,136.0 to 5,353.6 seconds with the
public default. The reduction was **25.0%, with a ±0.2 percentage point repeat-noise allowance**.
The allowance is the larger relative difference between the two unchanged baseline runs and the two
unchanged candidate runs, using each configuration's mean as denominator. Those differences were
0.135% and 0.185%. This measures repeatability on the fixed cohort.

| Measure                        | Baseline, run 1 | Candidate, run 1 | Baseline, run 2 | Candidate, run 2 |
| ------------------------------ | --------------: | ---------------: | --------------: | ---------------: |
| Total verification seconds     |         7,140.8 |          5,358.6 |         7,131.2 |          5,348.7 |
| Images building a solver model |              98 |               49 |              98 |               49 |
| Robust images                  |              42 |               43 |              42 |               43 |
| Checked adversarial examples   |               2 |               16 |               2 |               16 |
| Unresolved at the time limit   |              56 |               41 |              56 |               41 |

All images remain in the timing totals, including timeouts and changed verdicts. Across per-image
means, the candidate was faster by more than 1% on 55 images, slower by more than 1% on 5, and
within 1% on 40. The candidate/baseline ratio had 10th percentile 0.000105, median 0.676 and 90th
percentile 1.003. The [paired means](paired-means.csv), [summary and transitions](summary.json), and
[all 400 observations](runs/) preserve the individual results.

## Verification routes

The route counts were identical in both repeats. Images 74 and 78 took the original-input shortcut
in both configurations. Of the remaining 98 images, the candidate directly certified 35, found 12
adversarial examples at back-substitution corners, and found 2 through projected gradient descent
(PGD). It built a solver model for the remaining 49, certifying 8 and leaving 41 unresolved.

Fourteen images changed from a baseline timeout to a checked adversarial example. Image 77 changed
from a baseline timeout to a solver robustness certificate. Every definitive baseline verdict agreed
with the candidate. The public witness checks accepted all 36 saved witness observations across both
methods and repeats. Independent exact replay accepted all 36 saved inputs against the native
network and perturbation box. The audit also checked the full input domains of all 35 directly
certified images and all 49 solver fallbacks, covering 42,000 hidden-neuron intervals and 756 target
bounds. It matched all 397,000 affine coefficients and 510 biases against the native model.
[Exact audit](audit/README.md).

## Measurement

The [hosted run](https://github.com/vtjeng/MIPVerify.jl/actions/runs/46127914166) used 16 standard
`ubuntu-latest` runners, each measuring two lanes on separate physical cores. The runners exposed
AMD EPYC 7763, 9V45 and 9V74 processors, Intel Xeon Platinum 8573C processors, and Intel Xeon
6975P-C processors. Each image stayed on the same host and core for both methods and repeats. Method
order alternated by image and reversed on the repeat. [Host CPU records](hosts.json) preserve the
machine assigned to each pair of lanes.

Julia 1.12.6, HiGHS.jl 1.23.0 and the native HiGHS 1.14.0 binary were pinned by the
[shared environment manifest](../environment/Manifest.toml). Each worker used one Julia, BLAS and
HiGHS thread. Both configurations used the feasibility objective, all nine rival labels, a clipped
L-infinity pixel budget of Float64 `0.1`, Target LP tightening, a 20-second bound-LP limit and a
120-second final-MIP limit.

Before timing, every worker required the public API to certify RSL18a image 1 through
`:backward_bound`. Both methods were warmed on images 2001, 19 and 75. The
[setup timings](setup.csv) fall outside the per-image measurements. Each observation is the public
result's `:TotalTime`, covering screening, search, model construction, all bound LPs, the final MIP
and witness validation. Loading the model and data, warm-up and serializing CSV rows fall outside
that timer. The reported totals sum per-image elapsed times across different hosts and cores.

One hosted runner lost communication before publishing its logs or output. Its failed-only retry
completed the six affected images on a replacement runner. The failure and successful retry are
recorded in [issue #334](https://github.com/vtjeng/MIPVerify.jl/issues/334). All included
observations come from completed jobs, and each image's four observations come from the same
successful worker.

The comparison covers this fixed first-100-image cohort at the stated limits. The
[WK17a comparison](../README.md) uses a separate local machine and a larger cohort. Keep the public
screen enabled for these supported feasibility checks. On this RSL18a cohort it reduced time by more
than the measured repeat variation and settled 15 additional images. The production implementation
also maintains outward-rounded bounds and checked witnesses, which the accompanying tests and exact
audit verify.

## Reproduce

Use the repository revision containing this report. The hosted benchmark workflow and driver were
recorded at
[commit `1ee854ca480729823d4c5c3e5e1c78921de40ab6`](https://github.com/vtjeng/MIPVerify.jl/commit/1ee854ca480729823d4c5c3e5e1c78921de40ab6),
using
[production commit `26e928a`](https://github.com/vtjeng/MIPVerify.jl/commit/26e928a423aab28df03448b6f0aad37ad9f1650d).
Copy the shared manifest to the repository root, instantiate the benchmark workspace, and download
the model and data once before starting workers.

```sh
cp benchmarks/reports/2026-11-28-public-default/environment/Manifest.toml Manifest.toml
julia --project=benchmarks -e 'using Pkg; Pkg.instantiate(); using MIPVerify; MIPVerify.get_example_network_params("MNIST.RSL18a_linf0.1_authors"); MIPVerify.read_datasets("MNIST")'
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0 \
  julia --project=benchmarks benchmarks/reports/2026-11-28-public-default/run.jl \
  /tmp/rsl18a-public-rerun 0 1 RSL18a 100
```

For parallel measurements, assign distinct physical cores to the zero-based lanes and use the same
lane count in every invocation. Keep all four observations of an image on one worker. The archived
raw files use 32 lanes across the 16 hosts.

The [analysis script](analyze.py) validates exactly 100 images × 2 methods × 2 repeats with no
duplicate keys. It checks public witness flags, each saved input's size, lane consistency and
agreement of definitive verdicts before regenerating the summary and paired means.

```sh
python3 benchmarks/reports/2026-11-28-public-default/rsl18a/analyze.py
```
