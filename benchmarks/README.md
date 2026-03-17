# Benchmarks

Scripts for benchmarking MIPVerify on the MNIST WK17a network.

## `benchmark_wk17a_first100.jl`

Runs adversarial example search on MNIST test samples using the `MNIST.WK17a_linf0.1_authors`
network and writes per-sample results and aggregate metrics as CSV files.

### Usage

```sh
julia --project benchmarks/benchmark_wk17a_first100.jl \
  --out /tmp/bench-output \
  --samples 1:100 \
  --tightening interval_arithmetic \
  --main-time-limit 120 \
  --norm-order Inf \
  --log-level warn
```

### Arguments

| Argument            | Default        | Description                                                          |
| ------------------- | -------------- | -------------------------------------------------------------------- |
| `--out`             | **(required)** | Output directory for CSV results                                     |
| `--samples`         | `1:100`        | Sample indices (`start:stop`, `start:step:stop`, or comma-separated) |
| `--tightening`      | `mip`          | Tightening algorithm: `interval_arithmetic`, `lp`, or `mip`          |
| `--main-time-limit` | `120`          | Time limit in seconds for the main solve                             |
| `--norm-order`      | `Inf`          | Norm order for the perturbation (`Inf` or a number)                  |
| `--log-level`       | `warn`         | MIPVerify log level                                                  |

### Output

- `benchmark_per_sample.csv` — per-sample solve status, timing, objective value, and semantic
  outcome
- `benchmark_metrics.csv` — aggregate wall-clock time, summed solve times, status counts, and run
  metadata, including Julia version and dependency snapshot hash
- `dependency_versions.csv` — normalized resolved-package snapshot with package versions, tree
  hashes, source kind, and direct-dependency markers
- `dependency_manifest.toml` — copy of the active benchmark `Manifest.toml` for manual debugging
  (not consumed by any scripts)

## Nightly Benchmark Workflow

A GitHub Actions workflow (`.github/workflows/nightly-benchmark.yml`) runs the WK17a benchmark
nightly on 500 samples with `lp` tightening.

### Schedule

Runs daily at 6 AM UTC, or manually via `gh workflow run nightly-benchmark.yml`.

### Results storage

Results are committed to the
[`benchmark-results`](https://github.com/vtjeng/MIPVerify.jl/tree/benchmark-results) branch:

- **`tracking.csv`** — one row per nightly run with aggregate metrics (append-only)
- **`runs/YYYY-MM-DD/<run_id>/`** — immutable per-run artifacts for each nightly or rerun
  (`benchmark_metrics.csv`, `benchmark_per_sample.csv`, `dependency_versions.csv`,
  `dependency_manifest.toml`)

### `tracking.csv` columns

| Column                                        | Description                                            |
| --------------------------------------------- | ------------------------------------------------------ |
| `date`                                        | Run date (YYYY-MM-DD)                                  |
| `run_id`                                      | Immutable per-run identifier (`UTC timestamp` + SHA)   |
| `commit_sha`                                  | Git commit SHA benchmarked                             |
| `julia_version`                               | Julia version used for the benchmark                   |
| `dependency_snapshot_sha256`                  | SHA-256 hash of the normalized dependency snapshot     |
| `dependency_change_summary`                   | Text diff against the previous appended run's snapshot; `[no dependency changes]` when identical, missing when unavailable |
| `wall_clock_seconds`                          | Total wall-clock time for the benchmark run            |
| `sum_total_time_seconds`                      | Sum of per-sample total times                          |
| `sum_solve_time_seconds`                      | Sum of per-sample solve times                          |
| `median_solve_time_seconds`                   | Median per-sample solve time                           |
| `p90_solve_time_seconds`                      | 90th percentile per-sample solve time                  |
| `num_samples`                                 | Number of samples evaluated                            |
| `num_certified_no_adversarial_example`        | Samples proven robust (infeasible)                     |
| `num_adversarial_example_found_or_best_known` | Samples with adversarial examples found                |
| `num_time_limit_unresolved`                   | Samples that hit the time limit                        |
| `num_no_primal_solution_other`                | Samples with other non-primal outcomes                 |

## `append_to_tracking.jl`

Reads `benchmark_metrics.csv` and `dependency_versions.csv`, derives the dependency summary against
the previous run when available, and appends a summary row to `tracking.csv`. Used by the nightly
workflow.

### Usage

```sh
julia --project benchmarks/append_to_tracking.jl \
  --metrics-csv /tmp/bench/benchmark_metrics.csv \
  --dependency-versions-csv /tmp/bench/dependency_versions.csv \
  --tracking-csv /path/to/tracking.csv \
  --date 2024-01-15 \
  --commit-sha abc1234 \
  --run-id 2024-01-15T06-00-00Z-abc1234
```

## `compare_wk17a_benchmark.jl`

Compares two benchmark runs and gates on a maximum allowed regression in wall-clock and total solve
time.

### Usage

```sh
julia --project benchmarks/compare_wk17a_benchmark.jl \
  --baseline /tmp/bench-before \
  --candidate /tmp/bench-after \
  --max-regression 0.05
```

### Arguments

| Argument           | Default        | Description                                            |
| ------------------ | -------------- | ------------------------------------------------------ |
| `--baseline`       | **(required)** | Directory containing baseline `benchmark_metrics.csv`  |
| `--candidate`      | **(required)** | Directory containing candidate `benchmark_metrics.csv` |
| `--max-regression` | `0.05`         | Maximum allowed regression ratio (0.05 = 5%)           |

### Output

Prints a CSV-formatted comparison of wall-clock and total time, plus semantic outcome counts if
available. Exits with code 0 on `PASS` and 1 on `FAIL`.
