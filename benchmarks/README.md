# Benchmarks

Scripts for benchmarking MIPVerify on the MNIST WK17a network.

## `benchmark_wk17a_first100.jl`

Runs adversarial example search on MNIST test samples using the `MNIST.WK17a_linf0.1_authors`
network. It records formulation structure, progressive bound tightening, ReLU stability, and HiGHS
work in addition to solve outcomes.

### Usage

```sh
julia --project=benchmarks benchmarks/benchmark_wk17a_first100.jl \
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

- `benchmark_per_sample.csv` — per-sample solve outcome, timing, formulation structure, aggregate
  bound-tightening work, ReLU stability, and main-solver work
- `benchmark_relu_layers.csv` — one row per sample and ReLU layer, with layer shape, configured
  tightening algorithm, bounds and constraint-imposition timing (`bounds_time_seconds`,
  `constraint_time_seconds`), and stable or unstable counts
- `benchmark_tightening.csv` — one row per sample, ReLU layer, effective tightening algorithm, and
  bound direction; layer index `0` identifies bounds computed outside a ReLU layer
- `benchmark_metrics.csv` — aggregate wall-clock time, summed solve times, status counts, and run
  metadata, including Julia version and dependency snapshot hash
- `dependency_versions.csv` — normalized resolved-package snapshot with package versions, tree
  hashes, source kind, and direct-dependency markers
- `dependency_manifest.toml` — copy of the active benchmark `Manifest.toml` for manual debugging
  (not consumed by any scripts)

Inputs that the network already misclassifies have status `SKIPPED_PREDICTED_IN_TARGETED`. They do
not require a model or solve, but they count as zero-distance adversarial examples in the semantic
totals and have objective value and bound `0`.

### Per-sample instrumentation

`formulation_time_seconds` covers model construction, progressive tightening, target constraints,
the objective, and main-optimizer setup. `bound_tightening_time_seconds` is the sum of each ReLU
layer's complete bounds phase plus solver wall time for bounds computed outside a ReLU layer.
`formulation_residual_time_seconds` is formulation time minus that bound-tightening time. The
residual includes target and objective construction, optimizer setup, and any unscoped bound-loop
work that is not measured by a solver timer. `formulation_excluding_bound_solver_time_seconds`
subtracts only optimization-based bound-solver wall time, leaving interval propagation and
bound-loop overhead in the formulation time.

The two solver timing fields have different sources:

- `main_solve_wall_time_seconds` measures the wall time around the final `optimize!` call.
- `solve_time_seconds` is the final solve time reported by HiGHS.

The bound columns report logical requests, actual solver calls and statuses, progressive skips,
solver time, and work counters. ReLU columns use the formulation's four phase classes:
`zero_output`, `linear_in_input`, `constant_output`, and `split`. Stable count is the sum of the
first three; unstable count is `split`.

`num_structural_constraints` excludes variable bounds and integrality declarations.
`num_total_constraints` includes them. Main node, simplex-iteration, barrier-iteration, and relative
gap fields are missing when the solver does not expose a nonnegative value.

In `benchmark_tightening.csv`, `status_counts` and `skip_counts` contain sorted semicolon-separated
`name=count` pairs. Dedicated columns cover optimal and time-limit statuses and the four expected
progressive skip reasons.

`benchmark_schema_version` identifies the timing and output schema.
`semantic_outcome_schema_version` identifies the outcome-counting rules. Semantic schema 1, used by
historical runs through 2026-07-10, omitted already-misclassified skipped inputs from the
adversarial count. Semantic schema 2 includes them. Comparison tooling rejects runs with different
schema versions.

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
  (`benchmark_metrics.csv`, `benchmark_per_sample.csv`, `benchmark_relu_layers.csv`,
  `benchmark_tightening.csv`, `dependency_versions.csv`, `dependency_manifest.toml`)

### `tracking.csv` columns

| Column                                        | Description                                                                                                                |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `date`                                        | Run date (YYYY-MM-DD)                                                                                                      |
| `run_id`                                      | Immutable per-run identifier (`UTC timestamp` + SHA)                                                                       |
| `commit_sha`                                  | Git commit SHA benchmarked                                                                                                 |
| `benchmark_schema_version`                    | Version of the benchmark timing and output schema                                                                          |
| `semantic_outcome_schema_version`             | Version of the semantic outcome-counting rules                                                                             |
| `julia_version`                               | Julia version used for the benchmark                                                                                       |
| `dependency_snapshot_sha256`                  | SHA-256 hash of the normalized dependency snapshot                                                                         |
| `dependency_change_summary`                   | Text diff against the previous appended run's snapshot; `[no dependency changes]` when identical, missing when unavailable |
| `wall_clock_seconds`                          | Total wall-clock time for the benchmark run                                                                                |
| `sum_total_time_seconds`                      | Sum of per-sample total times                                                                                              |
| `sum_solve_time_seconds`                      | Sum of per-sample solve times                                                                                              |
| `median_solve_time_seconds`                   | Median per-sample solve time                                                                                               |
| `p90_solve_time_seconds`                      | 90th percentile per-sample solve time                                                                                      |
| `num_samples`                                 | Number of samples evaluated                                                                                                |
| `num_certified_no_adversarial_example`        | Samples proven robust (infeasible)                                                                                         |
| `num_adversarial_example_found_or_best_known` | Samples with adversarial examples found                                                                                    |
| `num_skipped_predicted_in_targeted`           | Already-misclassified inputs skipped before model construction; subset of adversarial outcomes                             |
| `num_time_limit_unresolved`                   | Samples that hit the time limit                                                                                            |
| `num_no_primal_solution_other`                | Samples with other non-primal outcomes                                                                                     |

## `append_to_tracking.jl`

Reads `benchmark_metrics.csv` and `dependency_versions.csv`, derives the dependency summary against
the previous run when available, and appends a summary row to `tracking.csv`. Used by the nightly
workflow.

### Usage

```sh
julia --project=benchmarks benchmarks/append_to_tracking.jl \
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
julia --project=benchmarks benchmarks/compare_wk17a_benchmark.jl \
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

Prints a CSV-formatted comparison of wall-clock and total time plus outcome counts. The gate
requires matching schemas, matching semantic partition counts, complete schema-2 partitions, and
timing regressions within the configured limit. It exits with code 0 on `PASS` and 1 on `FAIL`.
