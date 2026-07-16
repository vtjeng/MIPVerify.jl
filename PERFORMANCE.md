# Performance log

Measured performance changes, one row per PR, most recently merged first. Append a row in the same
PR that changes performance, with the measured impact and an evidence link. Numbers in a row come
from that row's paired evidence; rows are measured under different conditions and are not comparable
to each other.

## Verification (solve) performance

Changes to the time it takes to verify a sample: building the MIP model, tightening ReLU bounds
(interval arithmetic, LP, and MIP solves), and the main solve that proves robustness or finds an
adversarial example. This work scales with the number of samples verified. Columns, measured on the
paired 500-sample WK17a LP benchmark:

- **Total (before → after)**: end-to-end wall clock summed over samples where both runs reached the
  same outcome. Samples whose verdict or solve status differed (for example, hitting the 120 s time
  limit on one side only) are excluded, because their time ratio would measure the outcome change,
  not the speed change.
- **Median ratio**: per-sample after ÷ before; below 1 is faster.

| Date       | PR   | Change                                                         | Total (before → after) | Median ratio | Notes                                                                                                                                                                                                                | Evidence                                                                                                                                                                                      |
| ---------- | ---- | -------------------------------------------------------------- | ---------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-07-16 | #209 | Screen ReLU bounds progressively (interval, then LP, then MIP) | 6446 s → 2634 s        | 0.23         | The gain is entirely in bound tightening: 491/492 samples improved and the main solve is unchanged. Nightly confirmation pending — the 2026-07-17 run is the first to include it, together with the #220 solver pin. | [Report](https://github.com/vtjeng/MIPVerify.jl/pull/209#issuecomment-4989944753), [pair data](https://github.com/vtjeng/MIPVerify.jl/tree/benchmark-reports/pairs/2026-07-15-pr209-wk17a-lp) |
| 2026-07-13 | #208 | Certify LP tightening bounds from row duals                    | 3772 s → 4895 s        | 1.46         | A deliberate cost for soundness: bounds are recomputed from row duals instead of trusting solver-reported objectives. #209 later recovers more than this (net −40% vs pre-#208).                                     | [Retroactive report](https://github.com/vtjeng/MIPVerify.jl/pull/208#issuecomment-4988727305)                                                                                                 |

## CI-only improvements

Fixed per-run cost of the test and benchmark workflows. These do not scale with samples. Same-day
rows overlap in their before/after runs, so attribution between them is approximate.

- **Total (before → after)**: runner time summed over the jobs the row changes.
- **Long pole**: the longest single job of a full CI run; "—" where the row does not move it.

| Date       | PR   | Change                                            | Total (before → after) | Long pole     | Notes                                                                                                                                | Evidence                                                                                                                                                  |
| ---------- | ---- | ------------------------------------------------- | ---------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-07-16 | #237 | macOS test legs on native aarch64 Julia           | 1448 s → 707 s         | 729 s → 503 s | The macOS legs previously ran x64 Julia under Rosetta 2. Total covers the two macOS legs; the long pole moved from macOS to Windows. | Runs [before](https://github.com/vtjeng/MIPVerify.jl/actions/runs/29483424425) / [after](https://github.com/vtjeng/MIPVerify.jl/actions/runs/29536927187) |
| 2026-07-16 | #230 | Cache Julia package state for docs and benchmarks | 449 s → 299 s          | —             | Total covers the Documentation job (241 → 148 s) and the Benchmark Helper Tests job (208 → 151 s).                                   | Same before/after runs as #237                                                                                                                            |
| 2026-07-16 | #228 | Run notebook checks in one job                    | 1790 s → 477 s         | —             | Removes three of four setup-and-instantiate cycles. Wall clock also improved: the slowest of the four old jobs took 504 s.           | Same before/after runs as #237                                                                                                                            |
| 2026-07-16 | #235 | actionlint check for workflow files               | — → 7 s                | —             | A cost, not a saving: adds the actionlint job (with shellcheck) to every run.                                                        | [After run](https://github.com/vtjeng/MIPVerify.jl/actions/runs/29536927187)                                                                              |
| 2026-07-16 | #227 | Collect coverage on one CI leg                    | 3315 s → 2995 s        | 729 s → 611 s | Total covers the six test legs, from the PR's own paired runs; macOS was still x64 at the time.                                      | [Paired timing in PR](https://github.com/vtjeng/MIPVerify.jl/pull/227)                                                                                    |
| 2026-07-16 | #221 | Reduce default test suite runtime                 | 742 s → 199 s          | —             | Measured on a warm local run of the default suite; the CI test legs shrink with the same test selection.                             | [Measured impact in PR](https://github.com/vtjeng/MIPVerify.jl/pull/221)                                                                                  |
