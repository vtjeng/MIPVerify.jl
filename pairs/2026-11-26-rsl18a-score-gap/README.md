# Backward score-gap screening on RSL18a

This archive compares Target LP (`target_lp`) with the Backward score-gap screen (`screen`) on MNIST test images 1–100. Both search for any adversarial input within the clipped radius-0.1 pixel box, using the feasibility objective. RSL18a has one fully connected hidden layer of 500 ReLUs. The screen reuses its hidden-neuron ranges in the fallback MIP and validates returned inputs. Target LP uses the package's standard witness checks. Both use HiGHS with one thread, a 20-second limit per bound LP and a 120-second final-MIP limit.

## Measurement protocol

The Google Cloud N2-standard-96 host exposes an Intel Xeon CPU at 2.60 GHz with two sockets, 24 cores per socket and two hardware threads per core. The experiment uses its 48 exposed physical cores concurrently, with one worker per core. `run_cloud.py` derives one logical CPU for each unique `(socket, core)` pair from `lscpu`, then pins each worker to that CPU. It does not assign a second worker to the sibling hardware thread. All observations of an image run in the same process on the same core. CPU identity and topology are recorded in `results/cpu-info.txt` and `results/cpu-map.txt`. Timings reflect this concurrent workload, not an idle machine or the earlier local WK17a measurements.

Each image runs twice per method. Within each worker, the order alternates between Target LP/screen/screen/Target LP and screen/Target LP/Target LP/screen. The four rows per image record method and observation explicitly. The summary aligns these identifiers across files, averages each image's two observations for timing, and gives every image equal weight. Status and outcome counts are reported separately for both observations, without averaging categorical values.

Each process compiles the network layers and warms both methods on images 2001 and 2002 before timing its assigned images. These steps are excluded from per-image times and recorded in `setup-*.csv`. Timings include the complete verification call, including candidate-input checks. Before timing, the driver checks every compiled input coefficient against the native network's flattening permutation and weights. The 400 final timed observations are in `results/lane-*.csv`, across 48 worker files. All 400 expected keys, all 48 setup rows and the 49 fallback image IDs were checked. The fallback IDs exactly match the independently audited list. `data-check.json` records these checks. `results/summary.json` and `results/paired-results.md` were regenerated from the downloaded raw rows with this archive’s extended analyzer.

## Reproduction

Use [measured code revision ce20d25915988c3ddc929c8e5cd82778ee099e9f](https://github.com/vtjeng/MIPVerify.jl/commit/ce20d25915988c3ddc929c8e5cd82778ee099e9f). `environment/Manifest.toml` pins Julia 1.12.6, HiGHS.jl 1.23.0 and HiGHS_jll 1.14.0+0. Its MIPVerify entry points to `.` relative to the repository-root workspace manifest. Copy it to the repository root, not to `benchmarks/`. The authors' `two-layer.mat` weights have SHA-256 `e651f8ba5d14de0ade554d3b80ae287cf3609d499e91832893b318983bec20f2`.

The archived installer `environment/install-julia.sh` downloads Julia 1.12.6 and verifies its tarball against Julia's published SHA-256 list before extracting it under the user's home directory. The recorded installation reported `julia.tar.gz: OK` and `julia version 1.12.6`. Install the code checkout as `$HOME/code`, and set `study` to this archive's absolute path. From the code checkout:

```sh
cp "$study/environment/Manifest.toml" Manifest.toml
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$HOME/julia-1.12.6/bin/julia" --project=benchmarks -e 'using Pkg; Pkg.instantiate(); using HiGHS, MIPVerify; @assert VERSION == v"1.12.6"; @assert pkgversion(HiGHS) == v"1.23.0"'
cp "$study/run.jl" "$study/summarize.py" "$HOME/"
python3 "$study/run_cloud.py"
```

The runner expects `$HOME/output` to contain no `lane-*.csv` files. It determines worker count from physical topology, so another machine can use a different core count. Preserve its single-thread settings, warm-up and per-image method order. The runner calls `summarize.py` only after every worker completes.

To analyze downloaded results without rerunning measurements:

```sh
python3 "$study/summarize.py" "$study/results"
```

`summary.json` contains pooled timings, repeat differences, ratio quantiles, counts outside the ±1% band, removal of the 1/5/10 largest savings, largest savings/regressions, and a paired-image bootstrap (10,000 resamples, seed 20261126). The bootstrap measures sensitivity to the image mix. Repeat differences measure run-to-run variation separately. `paired-results.md` contains status/outcome counts and transitions by observation, plus within-method repeat transitions with image IDs. The analysis checks all 400 image/method/observation keys and reports opposing definitive outcomes across either observation as a failure.

## Independent numerical checks

`rational-bounds.log` records exact-rational checks of the earlier directly certified RSL18a images. `check_bounds.jl` calls the archived checker in `../2026-11-25-wk17a-backward-bounds/audit/check_hidden_bounds.jl`; it verifies hidden ranges and strictly negative score-gap bounds. The separate fallback check covers the 49 MIP-route images listed in `fallback-images.txt`, selected from the route scan. It passed all 24,500 hidden intervals (49 × 500) with zero unproved intervals, and all 441 target upper-bound enclosure assertions. Of those target bounds, 229 were nonnegative. The checker exited successfully. `rational-fallback-bounds.log` records stdout and `rational-fallback-summary.json` records these counts. In default mode every target bound must also be strictly negative. In `--fallback` mode the checker still requires each floating bound to enclose its exact counterpart, but permits nonnegative target gaps because these images proceed to MIP. Explicit source-match guards reject unexpected changes to the archived checker. From the measured code checkout:

```sh
julia --project=benchmarks "$study/check_bounds.jl" 1 4 6
julia --project=benchmarks "$study/check_bounds.jl" --fallback $(cat "$study/fallback-images.txt")
julia --project=benchmarks "$study/replay.jl" "$study/results"
```

`replay.jl` checks every recorded screen witness in the final CSVs against the native model using exact rational arithmetic. It checks the pixel box against both stored floating-point centers and original byte values divided by 255, and requires a nonnegative target-minus-true score gap. Replay ran successfully after timing. `results/witness-replay.csv` contains 32 checked rows across 16 distinct images, with every exact box and nonnegative-score-gap check true.

## Results

The mean per-image totals were 7,244.8 seconds for Target LP and 5,470.7 seconds for the Backward score-gap screen, a 24.5% reduction. The paired-image resampling range was 14.4–36.0% (95%, 10,000 resamples). This range describes image-mix sensitivity, not run-to-run noise. The unchanged Target LP repeat differed by 0.317%, and the screen repeat by 0.269%.

Both observations gave identical outcome counts: Target LP certified 42 images, returned two verified witnesses and left 56 unresolved; the screen certified 43, returned 16 verified witnesses and left 41 unresolved. Fourteen images changed from unresolved to a witness, and image 77 changed from unresolved to certified. No definitive outcomes conflicted, and neither method changed an image’s status or outcome between its repeats. The screen used 441 LP bound calls per observation versus 882 for Target LP.

`results/paired-results.md` provides the full status and semantic-outcome transition tables, including all affected image IDs.
