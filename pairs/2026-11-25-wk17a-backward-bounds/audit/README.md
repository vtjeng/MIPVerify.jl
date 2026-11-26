# Checking backward bounds and adversarial inputs

These scripts independently check the proposed floating-point bounds and stored adversarial inputs using rational arithmetic. Run them from the MIPVerify checkout used for this experiment, with its benchmark dependencies installed. Set `WK17_REPO` to the checkout path if running elsewhere. The scripts abort on a failed assertion. A successful run ends with exit code zero.

```sh
julia --project=benchmarks /path/to/audit/replay_witnesses.jl
# To check witness coordinates from a fresh benchmark CSV instead:
julia --project=benchmarks /path/to/audit/replay_witnesses.jl /path/to/benchmark_per_sample.csv
julia --project=benchmarks /path/to/audit/check_hidden_bounds.jl
# To prove one fallback image instead of the archived set of 34:
julia --project=benchmarks /path/to/audit/check_hidden_bounds.jl 584
julia --project=benchmarks /path/to/audit/check_affine_layers.jl
```

`confirmed-witnesses.csv` contains 126 saved adversarial inputs for images among the first 2,000 MNIST test images. `replay_witnesses.jl` evaluates the original network layers with exact rational weights and checks that each target score strictly exceeds the true-label score. It checks the radius-0.1 box against both the stored floating-point center and the original byte values divided by 255. It also checks every hidden preactivation against the bounds proposed by the backward verifier for that input.

`check_hidden_bounds.jl` checks the 34 archived fallback images by default, or a supplied list of image IDs. It independently proves that each proposed hidden interval encloses all values possible over the full allowed input box. Starting from exact affine interval sums, it uses a separate rational backward calculation when interval arithmetic alone is too weak. Each ReLU envelope is checked by its endpoints. It verifies both definitions of the input box and all nine upper bounds on target score minus true-label score. An unproved interval or bound aborts the script.

`check_affine_layers.jl` compares every compiled affine coefficient and bias against the original network layers on zero inputs and basis vectors. This covers convolution padding and the flatten permutation. The native network evaluation is exact in these particular tests because each output is one original weight or zero.

## Evidence from the research implementation

Files prefixed `research-` are outputs collected for research commit `5dd17c1`, the verifier that supplied the original two 2,000-image runs. They establish the following checks for that implementation.

- Every reused hidden interval on its 34 MIP fallback images was proved valid, totaling 163,336 intervals. Of their endpoints, 66,143 required a backward proof beyond simple interval arithmetic. All 306 target-score-difference upper bounds on those images passed the independent check.
- All 126 saved adversarial inputs passed exact replay and their 605,304 hidden-interval checks. The CSV includes directly found inputs and four MIP inputs refined after the timed runs to obtain strictly positive exact margins.
- The affine-layer test compared 7,533,672 coefficients and every bias with the original implementation.

`research-fallback-images.csv` lists the 34 checked images and their recorded outcomes. `research-summary.json` records the counts and witness-file hash. `research-hidden-certificates.csv`, `research-target-bounds.csv`, `research-witness-replay.csv`, and `research-affine-layers.txt` hold the measured outputs. The target-bound CSV displays rational bounds rounded to floating point for readability. Its checker performs each comparison in exact arithmetic.

The exact interval checks establish that reusing these bounds does not remove an allowed network state. MIP infeasibility verdicts still depend on the numerical solver's tolerances. The independent checks run outside timed benchmarks.

## Checks rerun on the follow-up implementation

`followup-witness-replay.csv` records a fresh successful replay of all 126 strict witnesses against the follow-up verifier, including all 605,304 hidden-interval checks. Its rows exactly match the earlier replay. `followup-affine-layers.txt` records a fresh successful check of all 7,533,672 affine coefficients, also identical to the earlier output.

The full-box check was rerun on all 34 fallback images. `followup-hidden-certificates.csv` records proofs of all 163,336 hidden intervals, with 66,143 endpoints requiring backward proofs and zero unproved intervals. `followup-target-bounds.csv` records all 306 independently checked upper bounds on target score minus true-label score. `followup-bounds-584.txt` retains the earlier individual rerun for image 584. `followup-summary.json` records this scope and the checked source-file hash.

These checks ran outside the per-image timing. Files prefixed `research-` preserve the corresponding older evidence for comparison.

## Checking the points returned during timing

The strict points above establish that the 126 images have adversarial inputs. The numerical solver may return different coordinates in a fresh run. To check the actual timed outputs, run these commands using the archived timed inputs. To regenerate those inputs, pass the raw output directory produced by the archive's `run.py`. The compact `runs/` timing CSVs omit input coordinates.

```sh
python3 /path/to/audit/crosscheck_timed_witnesses.py
# Or regenerate the archived input file from fresh run.py raw outputs:
python3 /path/to/audit/crosscheck_timed_witnesses.py --raw-output /path/to/raw
julia --project=benchmarks /path/to/audit/check_returned_points.jl
```

`timed-witnesses.csv.gz` preserves every timed adversarial input with its image ID, pass and verification route. The Python check compares each saved timed point with an already exact-checked input. It records every witness in `timed-witness-crosscheck.csv` and writes MIP points or differing coordinates to `timed-mip-witnesses.csv` for separate replay. The Julia check evaluates those unmodified coordinates in rational arithmetic and reports pixel-box membership and target score minus true-label score. Its `matches_inclusive_condition` column requires all pixel conditions and a score difference of at least zero. This can distinguish an exactly feasible point from one accepted within the solver's numerical tolerances.

The two fresh passes each used backward bounds for 1,844 images, found 122 inputs by projected gradient search, and sent the same 34 images to the MIP. `timed-route-summary.json` records both sets of fallback image IDs. They match the 34 images covered by the fresh full-box audit.

Across the two timed passes, all 244 gradient-search inputs exactly matched the already rational-checked coordinates. The eight unmodified MIP points came from images 212, 480, 952 and 1193, once per pass. Each exceeded the exact stored-center radius by about 3.33e−17; the maximum excess against the original byte center was 5.75e−17. For image 952, the exact target score minus true-label score was about −4.70e−13. All eight points passed the repository's numerical witness checks, and none met every exact condition. `timed-returned-points-replay.csv` records the individual checks and radius excesses.

The four separately saved inputs for those MIP images have strictly positive exact margins and satisfy both exact pixel boxes. Together with the 122 gradient-search inputs, they establish an adversarial input for each of the 126 reported images. The adjustments and additional checks ran outside the timing.
