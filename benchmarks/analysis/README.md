# Paired benchmark analysis

`analyze_pair.py` turns two `benchmark_wk17a_first100.jl` runs (a baseline and a candidate) into the
**distribution of per-sample improvements** — not just aggregate totals. It joins the two runs on
`sample_index` and emits a stats table plus plots.

The point: an aggregate like "−30% wall clock" hides the spread. The per-sample view shows whether
the change helps every sample or a few, whether the expensive samples benefit, and whether solve
statuses or semantic outcomes changed.

## Run

Managed with [uv](https://docs.astral.sh/uv/); dependencies are pinned in `uv.lock`.

    uv run analyze_pair.py \
      --baseline <baseline-run-dir> --candidate <candidate-run-dir> --out <out-dir> \
      --baseline-label "base <sha>" --candidate-label "candidate <sha>"

Each run dir must contain a `benchmark_per_sample.csv` (as written by
`benchmark_wk17a_first100.jl`).

## Outputs (into `<out-dir>`)

- `improvement_stats.md` / `.csv` — each run's objective, per-sample ratio distribution (min…max
  sweep, improved/regressed), aggregate saving + concentration, per-side solve-status counts, and
  status and semantic-outcome changes grouped by transition. Historical output without objective
  metadata used `closest`; cross-objective reports carry a prominent warning.
- `ratio_ecdf.png` — paired relative view: ECDF of per-sample `candidate/baseline` ratios, all
  series overlaid (dimensionless, so they share one axis).
- `absolute_runtime_ecdf.png`, `absolute_calls_ecdf.png` — per-side distributions (baseline vs
  candidate), grouped by unit.
- `magnitude_scatter.png`, `calls_scatter.png` — paired log-log candidate-vs-baseline scatter; below
  the `y = x` diagonal = candidate better.

The solver-call series and its plots appear only when both runs carry the instrumentation columns.

## Reading the numbers

- `ratio` = candidate ÷ baseline; `< 1` is faster/fewer.
- `pooled ratio` = candidate total ÷ baseline total — the aggregate counterpart to the per-sample
  `median`; the gap between them is the "aggregates hide the spread" story.
- `top-10 concentration` = the 10 samples with the largest absolute change as a share of the total
  absolute per-sample change (`0–100%`; high = a few samples dominate).
