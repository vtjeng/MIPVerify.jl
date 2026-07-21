# Benchmarks

Scripts for benchmarking MIPVerify on the MNIST WK17a network.

## PGD worst-margin warm-start experiment

The completed fixed-cohort result is in [PGD_WARMSTART_REPORT.md](PGD_WARMSTART_REPORT.md).

This experiment asks whether a strong non-adversarial PGD candidate helps branch and bound even when
PGD does not find an attack. For an input with true class `y`, PGD and the MIP optimize the same
worst margin:

```text
max_{x' in the L-infinity box} max_{j != y} f_j(x') - f_y(x')
```

A nonnegative, independently verified PGD margin is an attack and ends that sample before any MIP
solve. A negative margin is still useful as a candidate incumbent: values close to zero are the hard
near-misses this experiment is intended to test.

`PGDWarmStart.jl` is the executable source of truth for the treatments. The treatment names,
sources, coverage, and rotating block order are tested in `test/pgd_warm_start.jl`.

| Treatment     | Candidate source                         | Variables initialized                           | Role                    |
| ------------- | ---------------------------------------- | ----------------------------------------------- | ----------------------- |
| `cold`        | none                                     | none                                            | no-start control        |
| `random_full` | one deterministic uniform point per box  | every MIP variable after feasibility completion | complete-start control  |
| `pgd_full`    | PGD near-miss                            | every MIP variable after feasibility completion | candidate-quality test  |

Both complete treatments copy the tightened base model, fix its input to the selected candidate,
solve the resulting feasibility problem, and transfer the complete assignment (including
activations, ReLU phases, and maximum selectors) to a fresh copy of the base model. The random point
is not selected by margin. Its seed is the cached PGD sample seed plus `10_000_000`, which gives it
a separate deterministic random-number stream.

`original_sparse`, `pgd_sparse`, and `original_full` remain available through `--treatments` for
reproducing diagnostic runs. The sparse treatments are partial solver hints rather than complete
feasible incumbents and are not part of the primary comparison.

### Fixed protocol

Use the following protocol for the initial go/no-go experiment:

1. Generate candidates with deterministic batched PGD: epsilon `0.1`, step size `0.01`, 100 steps,
   20 restarts, and a recorded per-sample seed. Cache the exact candidate so every block uses the
   same point. Independently check its network output and perturbation membership. For solver
   starts, project this point inward by `max(1e-9, epsilon * 1e-8)` to avoid a floating-point value
   just outside a perturbation bound; keep the exact PGD point for attack verification.
2. Draw one uniform random control from the same perturbation box using its recorded seed. If PGD or
   the random control is a verified attack, persist the witness and skip all MIP treatments for that
   sample. An attack found by a MIP treatment is also persisted, but does **not** skip the remaining
   paired treatments.
3. Calibrate on one difficult near-miss for three blocks. Continue to the main cohort only if the
   full start completes and HiGHS reports or exhibits an initial incumbent from it.
4. Run a main cohort of 12 eligible samples: eight difficult cases and four controls. Select
   difficulty using only pre-treatment information (cached PGD margin and/or historical cold
   difficulty), and choose reserves before treatment results are inspected. Replace candidate
   attacks or originally misclassified inputs until the eligible stratum counts are met.
5. Run three complete paired blocks per sample. Rotate treatment order by block; do not change the
   treatment set or time limit within a block. Use LP tightening and a 30-second main-solve limit.
   If fewer than half of the eligible samples produce a resolved result in any treatment, rerun the
   entire cohort at 120 seconds instead of extending selected samples.
6. Run serially: one Julia process, one post-tightening base model, and at most one copied model at
   a time. Release the full-start completion model before treatment solves. Configure HiGHS with one
   thread, parallel mode off, seed zero, and identical options for all treatments. Refuse to start a
   new model when system-available memory is below 4 GiB.

The model formulation must be built and tightened once per sample, then copied for treatments. Every
copy must match the base model's variable count, binary count, structural-constraint count, and
variable-bound hash. This prevents treatment-specific formulation or tightening variance from being
mistaken for a warm-start effect.

### Required observations

Persist enough information to reproduce and audit every pair:

- sample, stratum, block, treatment order, PGD and random seeds, configuration, dependency snapshot,
  and model signature;
- exact PGD and random candidates, solver-start candidates, their margins, independent attack
  checks, and candidate-generation time;
- each full-start completion status and time, plus evidence that HiGHS accepted or used the start;
- termination and primal statuses, verified outcome, objective value and bound, incumbent and dual
  bound over time, nodes, simplex iterations, relative gap, and main-solve wall time;
- end-to-end treatment time, charging candidate generation and completion to the corresponding
  complete-start treatment; and
- process RSS before and after model copies and solves, peak RSS, and system-available memory.

Do not discard timeouts. Report their final bounds and work counters, and compare timeout rates at
the common limit. Treat wall time as a noisy secondary measure; simplex iterations are the primary
solver-work measure, with node count and bound trajectories as corroborating evidence.

### Decision rules and definition of done

For each eligible sample, first take the median across its three blocks. Across samples, summarize
paired ratios with a geometric mean and a sample-level bootstrap confidence interval. Use
`(candidate + 1) / (baseline + 1)` for simplex iterations so zero-work solves remain defined. Report
all three pairwise comparisons: `pgd_full / cold`, `random_full / cold`, and
`pgd_full / random_full`.

- **Solver-level support:** the geometric-mean simplex-iteration ratio is at most `0.90`, its 95%
  bootstrap upper bound is below `1.0`, and the fixed-limit timeout or final-bound results do not
  materially favor `cold`.
- **Promising but inconclusive:** the point estimate is at most `0.90`, but its interval crosses
  `1.0`, or solver work improves while noisy wall time does not.
- **Not supported:** the point estimate exceeds `0.90`, full starts are not reliably accepted, or
  fixed-limit progress is materially worse with `pgd_full`.
- **End-to-end practical:** report this separately; it requires the corresponding total-time ratio,
  including PGD and full-start completion, to be below `1.0`. Solver-level support does not imply
  this stronger result.

Attribute a supported result to PGD candidate quality only when `pgd_full / random_full` also has a
95% bootstrap upper bound below `1.0`. If `random_full` and `pgd_full` move together relative to
`cold`, interpret the result as a complete-start effect rather than a PGD-quality effect.

Before drawing a conclusion, compare fresh `cold` medians with a historical cold reference that uses
the same network, perturbation, objective, tightening, solver generation, and time limit. Flag any
sample outside the ratio range `[0.5, 2.0]` and investigate it. A result from a different objective
is not a comparable reference and must be labeled as such rather than used as a gate.

The experiment is complete only when the tests pass, the fixed cohort and all three paired blocks
are present (or have explicit candidate-attack skips), no memory guard was violated, the required
observations and dependency/configuration snapshot are archived, cold-reference discrepancies are
explained, and the report assigns one of the three solver-level conclusions plus the separate
end-to-end conclusion. The report must also say whether the random control supports attributing the
effect to candidate quality. Report limitations and the distribution of per-sample effects, not
only an aggregate runtime.

### Running the experiment

The scripts checkpoint CSV output after every candidate and treatment, so rerunning the same command
resumes rather than repeating completed solves. Keep a candidate cache unchanged after a benchmark
run starts; its hash is part of the run configuration.

```sh
julia --project=benchmarks benchmarks/generate_pgd_warmstart_candidates.jl \
  --out /tmp/pgd-warmstart/candidates.csv \
  --samples 19,46,246,479,359,407,444,233,404,432,194,122,313,460,4,32,428,280,36,468

julia --project=benchmarks benchmarks/select_pgd_warmstart_cohort.jl \
  --candidates /tmp/pgd-warmstart/candidates.csv \
  --out /tmp/pgd-warmstart/cohort.csv

julia --project=benchmarks benchmarks/benchmark_pgd_warmstart.jl \
  --out /tmp/pgd-warmstart/run \
  --candidates /tmp/pgd-warmstart/candidates.csv \
  --cohort-file /tmp/pgd-warmstart/cohort.csv \
  --blocks 1:3 --main-time-limit 30

julia --project=benchmarks benchmarks/summarize_pgd_warmstart.jl \
  --run /tmp/pgd-warmstart/run --mode cohort \
  --expected-samples 12 --expected-blocks 3
```

`benchmark_per_treatment.csv` is the paired result table, `benchmark_samples.csv` records PGD,
formulation, completion, and model-signature data, and `benchmark_trace.csv` contains incumbent and
bound progress callbacks. Solver logs are retained under `logs/`; these are also used to distinguish
accepted, rejected, and solver-completed starts. The summarizer writes its paired tables and report
under `analysis/` by default.

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
  --objective feasibility \
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
| `--objective`       | `feasibility`  | Adversarial-example objective: `feasibility` or `closest`            |
| `--norm-order`      | `Inf`          | Norm order for the perturbation (`Inf` or a number)                  |
| `--log-level`       | `warn`         | MIPVerify log level                                                  |

The benchmark defaults to `feasibility` because its primary series tracks fixed-budget robustness.
MIPVerify's public API still defaults to the exact minimum-distance `closest` objective. Pass
`--objective closest` to benchmark that objective.

The benchmark environment constrains HiGHS.jl to 1.23.x and HiGHS_jll to 1.14.x. Keep these
constraints aligned with the test environment so local, nightly, and paired measurements use the
same solver generation.

### Output

- `benchmark_per_sample.csv` — per-sample solve outcome, objective, target and perturbation witness
  checks, timing, formulation structure, aggregate bound-tightening work, ReLU stability, and
  main-solver work; `witness_output` and `perturbed_input_value` are semicolon-separated numeric
  arrays
- `benchmark_relu_layers.csv` — one row per sample and ReLU layer, with layer shape, applied
  tightening algorithm (`interval_arithmetic` when the layer has no nonconstant inputs, since
  constants need no bound solves), bounds and constraint-imposition timing (`bounds_time_seconds`,
  `constraint_time_seconds`), and stable or unstable counts
- `benchmark_tightening.csv` — one row per sample, ReLU layer, applied tightening algorithm, and
  bound direction; layer index `0` identifies bounds computed outside a ReLU layer
- `benchmark_metrics.csv` — aggregate wall-clock time, summed solve times, status and witness
  counts, and run metadata, including objective, Julia version, and dependency snapshot hash
- `dependency_versions.csv` — normalized resolved-package snapshot with package versions, tree
  hashes, source kind, and direct-dependency markers
- `dependency_manifest.toml` — copy of the active benchmark `Manifest.toml` for manual debugging
  (not consumed by any scripts)

Inputs that the network already misclassifies have status `SKIPPED_PREDICTED_IN_TARGETED`. They do
not require a model or solve, but they count as zero-distance adversarial examples in the semantic
totals and have objective value and bound `0`.

Only `INFEASIBLE` counts as a robustness certificate. `INFEASIBLE_OR_UNBOUNDED` does not identify
which condition the solver established, so the benchmark leaves it unresolved. It does not promote
the benchmark formulation's expected boundedness into a solver proof.

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
solver time, and work counters. Tightening rows name the algorithm used at that stage, so a layer
requested with MIP tightening can produce separate LP and MIP rows. ReLU columns use the
formulation's four phase classes: `zero_output`, `linear_in_input`, `constant_output`, and `split`.
Stable count is the sum of the first three; unstable count is `split`.

`num_structural_constraints` excludes variable bounds and integrality declarations.
`num_total_constraints` includes them. Main node, simplex-iteration, barrier-iteration, and relative
gap fields are missing when the solver does not expose a nonnegative value.

In `benchmark_tightening.csv`, `status_counts` and `skip_counts` contain sorted semicolon-separated
`name=count` pairs. Dedicated columns cover optimal and time-limit statuses and each progressive
skip reason.

`benchmark_schema_version` identifies the timing and output schema. Schema 6 records the
adversarial-example objective by name. Schema 5 splits witness verification into target and
perturbation checks. Schema 4 added verified-witness fields and solution- and objective-limit status
counts. Schema 3 recorded LP and MIP stages separately when progressive MIP tightening is requested.
`semantic_outcome_schema_version` identifies the outcome-counting rules. Semantic schema 4 requires
both the numeric target check and perturbation-family membership check before counting an
adversarial example. Semantic schema 3 first required a verified target witness and recorded failed
verification separately. Semantic schema 2 added already-misclassified skipped inputs to the
adversarial count. Comparison tooling rejects runs with different schema versions or objectives.
Metrics without objective metadata predate feasibility benchmarking and use `closest`.

## Nightly Benchmark Workflow

A GitHub Actions workflow (`.github/workflows/nightly-benchmark.yml`) runs the feasibility WK17a
benchmark nightly on 500 samples with `lp` tightening.

### Schedule

Runs daily at 6 AM UTC. Manual runs default to `feasibility`; select the closest-objective variant
with `gh workflow run nightly-benchmark.yml -f objective=closest`.

### Results storage

Results are committed to the
[`benchmark-results`](https://github.com/vtjeng/MIPVerify.jl/tree/benchmark-results) branch:

- **`tracking.csv`** — one row per nightly run with aggregate metrics (append-only)
- **`runs/YYYY-MM-DD/<run_id>/`** — immutable per-run artifacts for each nightly or rerun
  (`benchmark_metrics.csv`, `benchmark_per_sample.csv`, `benchmark_relu_layers.csv`,
  `benchmark_tightening.csv`, `dependency_versions.csv`, `dependency_manifest.toml`)

### `tracking.csv` columns

| Column                                         | Description                                                                                                                |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `date`                                         | Run date (YYYY-MM-DD)                                                                                                      |
| `run_id`                                       | Immutable per-run identifier (`UTC timestamp` + SHA)                                                                       |
| `commit_sha`                                   | Git commit SHA benchmarked                                                                                                 |
| `benchmark_schema_version`                     | Version of the benchmark timing and output schema                                                                          |
| `semantic_outcome_schema_version`              | Version of the semantic outcome-counting rules                                                                             |
| `adversarial_example_objective`                | `feasibility` or `closest`; missing historical values mean `closest`                                                       |
| `julia_version`                                | Julia version used for the benchmark                                                                                       |
| `dependency_snapshot_sha256`                   | SHA-256 hash of the normalized dependency snapshot                                                                         |
| `dependency_change_summary`                    | Text diff against the previous appended run's snapshot; `[no dependency changes]` when identical, missing when unavailable |
| `wall_clock_seconds`                           | Total wall-clock time for the benchmark run                                                                                |
| `sum_total_time_seconds`                       | Sum of per-sample total times                                                                                              |
| `sum_solve_time_seconds`                       | Sum of per-sample solve times                                                                                              |
| `median_solve_time_seconds`                    | Median per-sample solve time                                                                                               |
| `p90_solve_time_seconds`                       | 90th percentile per-sample solve time                                                                                      |
| `num_samples`                                  | Number of samples evaluated                                                                                                |
| `num_skipped_predicted_in_targeted`            | Already-misclassified inputs skipped before model construction; subset of adversarial outcomes                             |
| `num_certified_no_adversarial_example`         | Samples proven robust (infeasible)                                                                                         |
| `num_adversarial_example_found_or_best_known`  | Samples with adversarial examples found                                                                                    |
| `num_time_limit_unresolved`                    | Samples that hit the time limit                                                                                            |
| `num_no_primal_solution_other`                 | Samples with other non-primal outcomes                                                                                     |
| `num_witness_verification_failed`              | Samples with an available witness that failed either independent check                                                     |
| `num_witness_target_verification_failed`       | Available witnesses that failed the numeric network target or margin check; can overlap the perturbation failure count     |
| `num_witness_perturbation_verification_failed` | Available witnesses that failed perturbation-family membership; can overlap the target failure count                       |

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

## Paired before/after mini-reports

For a **performance-affecting change**, produce a paired before/after report that shows the
_distribution of per-sample improvements_ (not just aggregate totals) and attach it to the PR. The
workflow is three steps: run, analyze, publish.

### 1. Run both commits — `run_pair.sh`

Runs the WK17a benchmark on two commits, each in its own throwaway git worktree (so each uses that
commit's own `src` + `benchmarks`), then analyzes the pair. The current branch and working tree are
untouched. The runner develops each worktree's MIPVerify checkout into its benchmark environment and
applies the same HiGHS.jl 1.23.x / HiGHS_jll 1.14.x constraints to both sides, including older
commits whose benchmark project did not yet contain the pin.

```sh
benchmarks/run_pair.sh \
  --base <base-commit> --candidate <candidate-commit> \
  --out /tmp/pair-<slug> --samples 1:500 --tightening lp --main-time-limit 120 \
  --base-objective closest --candidate-objective feasibility
```

The side-specific objective flags are optional; without them, each commit uses its own benchmark
default.

Produces `/tmp/pair-<slug>/{base,candidate}` (benchmark outputs) and `/tmp/pair-<slug>/analysis`
(plots + tables). Write the filled report template to `analysis/report.md` before publishing.

### 2. Analyze — `analysis/`

`run_pair.sh` calls it for you; run it directly to re-analyze existing run dirs. It reports the
per-sample ratio distribution, aggregate saving and concentration, solve-status counts, and grouped
status and semantic-outcome changes, plus ECDF and scatter plots. See
[`analysis/README.md`](analysis/README.md).

### 3. Publish — `publish_report.sh`

Publishes an analysis dir to the **`benchmark-reports`** branch under a unique `pairs/<slug>/`. It
stages the analyzer's flat PNG files under `plots/`, keeps `report.md` and the statistics at the
archive root, and copies the sibling `base/` and `candidate/` runs as `baseline/` and `candidate/`.
The publisher is append-only and never forces: it aborts rather than overwrite an existing slug, and
retries a rejected push with fetch+rebase, so nothing already on the branch can be clobbered.

```sh
benchmarks/publish_report.sh /tmp/pair-<slug>/analysis <YYYY-MM-DD-slug>
```

After a successful push, the publisher prints the pinned raw URL base used by the report template's
`{{pinned-raw-base}}` plot links.

Then post a PR comment following [`REPORT_TEMPLATE.md`](REPORT_TEMPLATE.md): preamble (what the PR
changes, benchmark setup, link to the published `pairs/<slug>/` folder), `## Summary`, then
`## Detailed statistics` with `### Plots` as its first subsection — plots come before every table so
readers get the shape of the distribution before the numbers.

`benchmark-reports` is a manual, human-published branch, separate from the CI-managed
`benchmark-results` branch — the two never share a path.
