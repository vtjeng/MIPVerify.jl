# Performance log

Measured performance changes, one row per PR, newest first. Append a row in the same PR that changes
performance, with the measured impact and an evidence link. Numbers in a row come from that row's
paired evidence; rows are measured under different conditions and are not comparable to each other.

## Verification (solve) performance

Changes to solve and tightening work. These scale with the number of samples verified. Total is
end-to-end wall clock pooled over the paired 500-sample WK17a LP benchmark (comparable samples
only); median ratio is per-sample candidate ÷ baseline, below 1 is faster.

| Date       | PR   | Change                                                         | Total (base → cand) | Median ratio | Notes                                                                                                                                      | Evidence                                                                                                                                                                                      |
| ---------- | ---- | -------------------------------------------------------------- | ------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-07-16 | #209 | Screen ReLU bounds progressively (interval, then LP, then MIP) | 6446 s → 2634 s     | 0.23         | 491/492 samples improved; main solve unchanged (pooled ratio 1.00). First post-merge nightly (2026-07-17) pending confirmation.            | [Report](https://github.com/vtjeng/MIPVerify.jl/pull/209#issuecomment-4989944753), [pair data](https://github.com/vtjeng/MIPVerify.jl/tree/benchmark-reports/pairs/2026-07-15-pr209-wk17a-lp) |
| 2026-07-13 | #208 | Certify LP tightening bounds from row duals                    | 3772 s → 4895 s     | 1.46         | Soundness fix: stops trusting solver-reported objectives. Nightly rose from 52–63 to 78–82 min. #209 pays this back: net −40% vs pre-#208. | [Retroactive report](https://github.com/vtjeng/MIPVerify.jl/pull/208#issuecomment-4988727305)                                                                                                 |

## CI-only improvements

Fixed per-run cost of the test and benchmark workflows. These do not scale with samples. Total is
runner time summed over the jobs the row changes; long pole is the longest single job of a full CI
run, "—" where the row does not move it. Same-day rows overlap in their before/after runs, so
attribution between them is approximate.

| Date       | PR   | Change                                            | Total (before → after) | Long pole     | Notes                                                                                      | Evidence                                                                                                                                                  |
| ---------- | ---- | ------------------------------------------------- | ---------------------- | ------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-07-16 | #237 | macOS test legs on native aarch64 Julia           | 1448 s → 707 s         | 729 s → 503 s | Previously x64 under Rosetta 2; the long pole moved from macOS to Windows.                 | Runs [before](https://github.com/vtjeng/MIPVerify.jl/actions/runs/29483424425) / [after](https://github.com/vtjeng/MIPVerify.jl/actions/runs/29536927187) |
| 2026-07-16 | #235 | actionlint check for workflow files               | — → 7 s                | —             | New check: a cost, not a saving.                                                           | [After run](https://github.com/vtjeng/MIPVerify.jl/actions/runs/29536927187)                                                                              |
| 2026-07-16 | #230 | Cache Julia package state for docs and benchmarks | 449 s → 299 s          | —             | Documentation 241 → 148 s; Benchmark Helper Tests 208 → 151 s.                             | Same before/after runs as #237                                                                                                                            |
| 2026-07-16 | #228 | Run notebook checks in one job                    | 1790 s → 477 s         | —             | Four jobs → one; the slowest previous notebook job was 504 s, so wall clock also improved. | Same before/after runs as #237                                                                                                                            |
| 2026-07-16 | #227 | Collect coverage on one CI leg                    | 3315 s → 2995 s        | 729 s → 611 s | Six test legs, from the PR's own paired runs (pre-#237, so macOS still x64).               | [Paired timing in PR](https://github.com/vtjeng/MIPVerify.jl/pull/227)                                                                                    |
| 2026-07-16 | #221 | Reduce default test suite runtime                 | 742 s → 199 s          | —             | Warm local run of the default suite; CI test legs shrink correspondingly.                  | [Measured impact in PR](https://github.com/vtjeng/MIPVerify.jl/pull/221)                                                                                  |
