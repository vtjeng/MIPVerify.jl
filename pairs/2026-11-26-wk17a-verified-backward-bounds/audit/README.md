# Exact checks of the returned adversarial inputs

The corrected verifier checks its returned inputs against the exact radius-0.1 pixel box and an inclusive target-score condition. This audit evaluates the actual coordinates written by the timed benchmark. It uses the original network layers with rational weights and pixels, independently of the verifier's compiled sparse representation.

From the MIPVerify checkout with its benchmark environment installed, run:

```sh
julia --project=benchmarks /path/to/audit/check_returned_points.jl
```

The default input is `timed-witnesses.csv.gz`. `timed-returned-points-replay.csv` records each point's image ID, pass and route, membership in [0,1], the radius check against both the stored center and the original bytes divided by 255, and the exact target score minus true-label score. The inclusive condition requires that difference to be at least zero, so a genuine tie remains valid. The rational score difference is included alongside its floating-point display value. A failed exact check makes the script exit unsuccessfully.

Repeated coordinates for the same image share one exact calculation. The script still writes one result per timed record. The cache key includes the complete saved coordinate string, so a changed coordinate receives a fresh calculation.

To regenerate the compressed inputs from the raw directories produced by `run.py`:

```sh
python3 /path/to/audit/extract_timed_witnesses.py --raw-output /path/to/raw
```

`--ablation --raw-output /path/to/ablation` instead extracts the runs that isolate reuse of the hidden bounds. Pass the resulting `ablation-timed-witnesses.csv.gz` to the Julia checker to replay those points too.

## Results on the corrected timed runs

Both primary passes returned 126 adversarial inputs, for 252 actual timed records. Every record passed the independent exact checks against both pixel-box definitions and the original network. All score differences were strictly positive, with a minimum of about 5.40e−15. The verifier used its outward interval check for 246 returns and exact rational evaluation for six. Eight returned records had been corrected from a MIP proposal within the timed verification call. No saved point was substituted after timing.

The bound-reuse comparison returned another 250 points, all of which passed the exact checks. Without reuse, points for images 480 and 952 could not be verified and were withheld with the outcome `witness_verification_failed` and empty coordinate fields. These outcomes remain in the raw ablation data and its solve-status analysis.

`summary.json` records the counts, routes and file hashes. `exact-evaluations.csv` records the 128 distinct image/coordinate pairs evaluated with the native rational network. Exact coordinate identity links those calculations to all 502 primary and ablation records. Each result file retains one row per actual returned point. The main results are in `timed-returned-points-replay.csv`; the comparison results are in `ablation-returned-points-replay.csv`.

To share the exact-calculation cache while replaying both input archives:

```sh
julia --project=benchmarks /path/to/audit/check_returned_points.jl \
  /path/to/audit/timed-witnesses.csv.gz \
  /path/to/audit/ablation-timed-witnesses.csv.gz
```

## Connection to the earlier bounds proof

The witness correction leaves the proposed hidden bounds and input boxes unchanged. `bound-values-comparison.csv` compares every hidden interval on the same 34 MIP fallback images before and after the correction, totaling 163,336 intervals. The paired SHA-256 hashes are identical. All input boxes and target-score-difference bounds also match, and the compiled affine coefficients and biases are equal. `bound-values-summary.json` records the source hashes and image IDs.

The previous source was commit `3745c58`, with SHA-256 `cf142faf795bc27fbbff56cdeb02a85b98a1c9d3bbea7dda497d998904ef7b62`, the same file checked by the earlier exact audit. That audit independently proved all 163,336 hidden intervals and all 306 target bounds, and checked every compiled affine coefficient against the original layers. [Earlier exact proofs and their outputs](https://github.com/vtjeng/MIPVerify.jl/tree/75135e0bd6ef43f61c9d4285f803aa5484dc82b5/pairs/2026-11-25-wk17a-backward-bounds/audit) · [Previous verifier source](https://github.com/vtjeng/MIPVerify.jl/blob/3745c5845a0126a1fff8aa564d3c0fb9e2957f7c/benchmarks/backward_margin_screen.jl)

To repeat the array comparison, supply the previous and corrected verifier source files from their respective checkouts:

```sh
julia --project=benchmarks /path/to/audit/compare_bound_values.jl \
  /path/to/previous/benchmarks/backward_margin_screen.jl \
  /path/to/corrected/benchmarks/backward_margin_screen.jl
```

The comparison uses the same 34 IDs. Both corrected primary passes sent exactly those 34 images to the MIP, as recorded in `summary.json`. The exact array comparison connects the corrected implementation to the existing full-box proof. Final MIP infeasibility verdicts still rely on the solver's numerical tolerances.
