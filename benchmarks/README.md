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
  metadata

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
