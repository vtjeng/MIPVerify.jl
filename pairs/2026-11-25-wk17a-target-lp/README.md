# WK17a output-margin screening

Each subdirectory under `runs/` is the repository benchmark output for a single run on a pinned physical CPU core. The four lanes contain disjoint, interleaved images. Every image runs on the same core for both methods. The first 500 images have two runs per method, with opposite orders on alternating lanes. The fresh 1,500 images have one run per method, again reversing order on alternating lanes. Times are the sums of per-image `total_time_seconds`. The 2,000-image statistic weights each image once, averaging the two observations where available.

The solver is HiGHS.jl 1.23.0 with native HiGHS 1.14.0+0, on Julia 1.12.6 and an Intel i7-8700K. The historical repository pair runner pins this solver combination. The benchmark wrapper sets Julia, BLAS and solver threads to one. It uses the original benchmark CLI and first runs image 2001 to warm up each fresh Julia process. The warm-up is outside the cohort. Model formulation, solver work, and any additional compilation inside the measured calls remain included.

To reproduce on Linux, create separate checkouts of baseline `144974b` and candidate `80ef0fe` (the published commit with the same verifier and benchmark code as the measured checkout `9385afd`). Copy a run’s archived `dependency_manifest.toml` into each checkout as `Manifest.toml`. Run `julia --project=benchmarks -e 'using Pkg; Pkg.instantiate()'` there. The manifest refers to MIPVerify by the relative path `.` in Julia’s root workspace. Set `WK17_BASE_TREE` and `WK17_TARGET_TREE` to those absolute paths. The runner assigns cores 0, 2, 4 and 6. Adjust the affinity list for your machine. Direct the output to an empty directory with `WK17_OUTPUT`. The defaults preserve existing completed run directories.

```sh
python3 run.py 1 500 4
python3 run.py 501 2000 2
```

To analyze the archived results, run `python3 analyze.py all`. The analysis checks sample alignment and conflicting definitive outcomes before comparing time. It reports each repeat, the all-resolved intersection, and savings after removing the 1, 5, or 10 largest favorable contributions. Its bootstrap resamples complete paired images with seed 20261125. Bootstrap intervals describe image mix in this fixed workload. The LP repeat measures run-to-run noise separately.

The larger run started only after both target-LP totals were below both LP totals, the mean savings exceeded the larger within-method repeat difference, the savings stayed positive after removing the five largest gains, and the mean resolved count was at least the baseline’s. `gate-result.txt` records these checks. For coverage on 2,000 images, `coverage.txt` lists both directions of disagreement in the first observation per image, plus the intersection where both methods built models.

The [five figures](plots/) compare per-image means for the two methods. End-to-end time covers all 2,000 images. The phase and call plots cover the 1,951 images that required a model. Run `python3 plots/generate.py` with NumPy, pandas and Matplotlib to regenerate the figures and their [summary statistics](plots/summary.json). The [score bounds for five example images](example-score-bounds.csv) give the interval and screened upper bounds on the target score minus the true-label score.
