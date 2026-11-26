# WK17a backward bounds before exact validation of returned inputs

This experiment measured the first backward verifier in PR #327. Eight MIP-returned points passed solver tolerances but exceeded the exact pixel budget. Exact replay confirmed separate valid points for all four affected images. The [point-by-point audit](audit/README.md) records this discrepancy.

The benchmark runs Target LP and the backward verifier on MNIST test images 1–2,000. Each pixel can change by at most 0.1 within [0, 1]. A target that ties the true label is feasible. Both methods use one HiGHS thread, a 20-second limit on each bound LP, and a 120-second limit on the final mixed-integer solve. Each total is the sum of per-image verification times. Julia startup, loading the stored network and warm-up on image 2001 are outside those totals.

## Run the comparison

Check out baseline [8e1d435](https://github.com/vtjeng/MIPVerify.jl/commit/8e1d435ea48b3ad4f750a2ebec28f166248ec61e) and candidate [6b8019e](https://github.com/vtjeng/MIPVerify.jl/commit/6b8019e51a728c8329f327514c75b3dff1717304) in separate directories. These trees are identical to measured revisions `cd77fe9` and `865d0ff` after #326 was squash-merged. Both use the [archived project environment](environment/). Copy `Manifest.toml` to the root of each checkout and run `julia --project=benchmarks -e 'using Pkg; Pkg.instantiate()'` there. Julia 1.12.6 loaded HiGHS.jl 1.23.0 and HiGHS 1.14.0+0 on an Intel Core i7-8700K. MIPVerify downloads the authors’ WK17a weights and MNIST test set on first use.

Run the following commands from this evidence directory on Linux. The first script uses the [Target LP harness](../2026-11-25-wk17a-target-lp/driver.jl) beside this archive. Set the checkout paths and choose a new output directory. Cores 0, 2, 4, and 6 are separate physical cores on the measurement machine, and core 8 runs the independent reuse comparison. Adapt these CPU IDs to your machine if necessary.

```sh
python3 run.py --baseline /path/to/baseline --candidate /path/to/candidate \
  --raw /tmp/backward-raw --runs /tmp/backward-runs
python3 compare_reuse.py --candidate /path/to/candidate --core 8 \
  --raw /tmp/reuse-raw --output /tmp/reuse-comparison.csv
```

Each image and its repeats stay on one core. Two cores run Target LP, backward bounds, backward bounds and Target LP in that order. The other two reverse the order. Each run starts a fresh process. The separate comparison runs the same backward verifier with reuse disabled and enabled on every image, alternating method order between odd and even image IDs. The [setup and warm-up records](environment/setup-times.csv) give each process’s warm-up time on image 2001 and, for the candidate, the time to compile the network layers. These timings exclude Julia startup and are outside the paired per-image times.

## Read the results

The [per-run CSVs](runs/) use the method, observation (1 or 2) and lane (0–3) in each filename. [Per-image means](paired-means.csv) retain zero solver costs on images resolved without a MIP. [Paired statuses](paired-statuses.csv) include the solver status, verification outcome and route for each image and observation. `NO_SOLVER` identifies a result from backward screening or verified witness search. `SKIPPED_PREDICTED_IN_TARGETED` identifies an image the baseline already misclassified before building its model.

Run `python3 analyze.py` with NumPy, pandas and Matplotlib to regenerate the [five figures](plots/), [timing tables](metrics.md), [status transitions](transitions.md) and [summary](summary.json) from the archived runs. For a fresh run, use a copy of this directory with your newly generated CSVs under `runs/`. The analysis averages repeats within each image and pairs by MNIST test index. Status counts and transitions use the first observation. The bootstrap resamples pairs of images. Ratios omit only zero denominators, and the absolute comparisons retain true zeros.

Brief package precompilation on the shared host overlapped part of the repeated runs. The repeat differences and sensitivity to dropping the ten largest savings are reported in [the timing tables](metrics.md).

[Reuse comparison, paired by image](reuse-comparison.csv) · [Exact bound and witness checks](audit/README.md)
